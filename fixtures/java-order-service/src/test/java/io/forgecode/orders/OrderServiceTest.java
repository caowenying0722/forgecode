package io.forgecode.orders;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.math.BigDecimal;
import java.util.Arrays;
import java.util.Collections;

import org.junit.jupiter.api.Test;

class OrderServiceTest {
    private final OrderService service = new OrderService();

    @Test
    void multipliesUnitPriceByQuantity() {
        OrderItem item = new OrderItem(3, BigDecimal.valueOf(1250, 2));

        assertEquals(
            BigDecimal.valueOf(3750, 2),
            service.calculateTotal(Arrays.asList(item))
        );
    }

    @Test
    void emptyOrderHasZeroTotal() {
        assertEquals(
            BigDecimal.ZERO,
            service.calculateTotal(Collections.<OrderItem>emptyList())
        );
    }
}
