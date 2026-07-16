package io.forgecode.orders;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.math.BigDecimal;
import java.util.Arrays;

import org.junit.jupiter.api.Test;

class OrderServiceHiddenTest {
    private final OrderService service = new OrderService();

    @Test
    void totalsMultipleItemsWithDifferentQuantities() {
        assertEquals(
            BigDecimal.valueOf(5498, 2),
            service.calculateTotal(Arrays.asList(
                new OrderItem(2, BigDecimal.valueOf(1999, 2)),
                new OrderItem(3, BigDecimal.valueOf(500, 2))
            ))
        );
    }

    @Test
    void zeroQuantityContributesNothing() {
        assertEquals(
            BigDecimal.valueOf(0, 2),
            service.calculateTotal(Arrays.asList(
                new OrderItem(0, BigDecimal.valueOf(9999, 2))
            ))
        );
    }
}
